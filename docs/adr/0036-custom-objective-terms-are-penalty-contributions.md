---
status: accepted
---

# Custom objective terms are penalty contributions, not objective-expression mutations

## Context

`cobj` and `ccnt` on a [`JuMPOptimiser`](../../src/20_Optimisation/10_JuMPOptimiser.jl) are the
user-facing escape hatch: subtype `CustomJuMPObjective` / `CustomJuMPConstraint` and implement one
builder method. The two hooks were meant to be siblings. They had drifted badly apart:

```julia
add_custom_constraint!(model, ccnt, opt, attrs)                              # 4 args
add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)  # 7 + varargs
```

The objective side's extra arity was not carrying extra information — it was carrying
*redundancy*, and hiding three defects inside it.

1. **`cobj` was derivable from its neighbour.** All nine `set_portfolio_objective_function!` call
   sites passed the same pair: `X.opt.cobj` as `cobj` and `X` as `opt`. The fourth argument was a
   function of the fifth, always.

2. **`pret` and `pr` duplicated the bundle.** The constraint side passes `attrs` and lets the
   implementer read `attrs.ret` / `attrs.pr`. The objective side hand-threaded those same two
   values as separate positionals — and on the `MeanRisk` path `attrs` was *already* arriving at
   the hook anyway, undocumented, as the tail of `args...`.

3. **The `opt` slot had no single type.** `set_portfolio_objective_function!` declared
   `opt::JuMPOptimisationEstimator` and received `mr::MeanRisk`, but `NearOptimalCentering`
   bypassed it and called the hook directly with a `JuMPOptimiser` — a *different branch* of the
   type hierarchy (`<: BaseJuMPOptimisationEstimator`). A hook that dispatched on `opt`, or reached
   into `opt.opt`, worked everywhere except under NOC. The untyped `opt` parameter is what let this
   hide; the arity is what let it go unnoticed. `test_03c` had come to *key its NOC regression
   detector on the inconsistency itself*.

Two further problems were structural rather than cosmetic:

**A quadratic custom term was impossible on some configurations.** The hook received `obj_expr` and
mutated it in place via `JuMP.add_to_expression!`. Whether `obj_expr` is an `AffExpr` or a
`QuadExpr` depends on the risk measure and objective — `MaximumReturn` yields affine,
`MinimumRisk` + `Variance` quadratic — and `add_to_expression!(::AffExpr, ::QuadExpr)` is a
`MethodError`. Promotion allocates a *new* expression, so a hook returning `nothing` had no way to
hand it back. `add_penalty_to_objective!` had already solved this for the library's own penalties,
and says so in its docstring; the user hook was the one path denied the fix.

**The sign was the user's problem, and under `MaximumRatio` it had no answer.** Because the hook
was handed the raw `obj_expr`, the implementer had to know the optimisation *sense* to orient their
term — subtract to reward under a minimisation, add under a maximisation. The documented worked
example carried a `reward_sign` dispatch table and a boxed warning for this, plus an explicit
`MaximumRatio` carve-out: that problem is solved through a homogenising transform that lands in
*either* a maximisation or a risk-minimisation form depending on the risk measure, so no fixed sign
is correct and a tilt against it deliberately raised a `MethodError`.

## Decision

**A custom objective term contributes to the objective penalty accumulator; it never touches the
objective expression.** The hook runs *before* `add_penalty_to_objective!`, so the sense-correct
factor the library already applies to its own penalties applies to user terms too.

```julia
set_portfolio_objective_function!(model, obj, optimiser, attrs)   # was 6 + varargs
add_custom_objective_term!(model, obj, cobj, optimiser, attrs)    # was 7 + varargs
add_custom_constraint!(model, ccnt, optimiser, attrs)             # unchanged
```

1. **`obj_expr` leaves the hook signature.** The implementer calls
   `add_to_objective_penalty!(model, expr)`, which already handles `AffExpr → QuadExpr` promotion
   internally. Quadratic terms work on every configuration. A *reward* is a negative penalty.

2. **The sense is the library's problem.** Every builder folds the accumulator in with a factor
   that is already correct for its own sense (`+1` into a `Min`, `-1` into a `Max`, including both
   `MaximumRatio` branches). A term contributed to the accumulator penalises under all six
   objective forms; `reward_sign` and the `MaximumRatio` carve-out are deleted, not relocated.

3. **`attrs` replaces `pret` and `pr`.** Verified value-preserving on every path:
   `jump_optimiser_from_attributes` sets `ret = attrs.ret` and `pe = attrs.pr`, so NOC's
   `opt.ret` / `opt.pe` *were* `attrs.ret` / `attrs.pr`; `MeanRisk` threads both down unchanged.
   `attrs` is now forwarded through `solve_noc!` to the constrained NOC builder, which previously
   had no bundle to reach into — which is precisely why it hand-passed the two values.

4. **`optimiser` always means the outer optimisation estimator** (`mr`, `noc`, `rb`) — the meaning
   the constraint hook already kept. NOC passes `noc`, not its flattened `JuMPOptimiser`. The
   parameter is renamed from `opt` to `optimiser` because `assemble_jump_model!` already uses `opt`
   for the `JuMPOptimiser` sitting beside it, and the collision was one line wide.

5. **`cobj` is dropped from `set_portfolio_objective_function!`** — it was never dispatched on
   there, only forwarded, and is `optimiser.opt.cobj` on every path. `obj` stays in both signatures:
   the builder genuinely dispatches on it to decide which objective to construct, and the hook
   exposes it so a term can vary by objective.

6. **`args...` is dropped.** After `attrs` becomes explicit the tail carried only `fees`, which is
   `attrs.fees`. `ProcessedJuMPOptimiserAttributes` is the extension point: a future argument is a
   new field, not a new positional.

7. **Both no-op fallbacks are typed on their supertypes.** A `CustomJuMPObjective` or
   `CustomJuMPConstraint` with no builder method now raises, rather than silently contributing
   nothing. The untyped `f(args...; kwargs...)` catch-alls absorbed *any* mis-shaped call — the
   mechanism by which a `ccnt` vector once silently no-op'd, and the mechanism by which this very
   signature change would otherwise have turned every existing hook into a silent no-op returning a
   plausible, wrong portfolio.

8. **The extension surface is marked `public`, not exported.** `CustomJuMPObjective`,
   `CustomJuMPConstraint`, `add_custom_objective_term!`, `add_custom_constraint!`,
   `add_to_objective_penalty!`, `ProcessedJuMPOptimiserAttributes`, and the model accessors are
   public API and documented as such, without putting `get_w` / `get_k` into the global namespace.

## Considered options

- **Return `obj_expr` from the hook** so the implementer can promote it, mirroring
  `add_penalty_to_objective!`. Rejected: it unlocks quadratic terms but leaves the sign footgun and
  the `MaximumRatio` hole exactly where they were, and it adds a new one — a hook that forgets to
  `return obj_expr` returns `nothing`, which the caller must then guard against.
- **Give the hook the *declared* objective (`optimiser.obj`) instead of the per-solve one.**
  Rejected on two counts. `RiskBudgeting` and `RelaxedRiskBudgeting` have no `obj` field at all —
  they are structurally minimum-risk — so it needs a `declared_objective` accessor plus a fallback
  convention for estimators that have no objective to declare. And it would tell the hook
  `MaximumUtility` while constrained NOC is building a min-*distance* model. The per-solve
  objective always describes the model actually being built. The cost, accepted knowingly: a
  `Frontier` sweep dispatches the hook three ways within one `optimise` call (`MinimumRisk` and
  `MaximumReturn` for the endpoint sub-solves, then the declared objective per frontier point), and
  because those endpoints set the sweep's bounds, an objective-dependent term shifts where the
  frontier begins and ends.
- **A dedicated context struct** (`add_custom_objective_term!(model, cobj, ctx)`). Rejected: it
  invents a second bundle alongside `ProcessedJuMPOptimiserAttributes`, and two bundles are worse
  than one.
- **Keep a deprecation shim** matching the old 7-argument order. Rejected: on `0.x`, breaking
  changes ship in a minor bump, and a shim that forwards the old call cannot reproduce the old
  *semantics* — the old hook mutated `obj_expr` directly, which no longer happens at all. A shim
  would have to be a lie or a no-op.
- **Fix only the objective fallback, leaving the constraint one untyped.** Rejected: it would ship
  a release where `cobj` fails loudly and `ccnt` fails silently — a fresh asymmetry introduced by a
  refactor whose purpose was removing asymmetry, on the side with the worse track record.

## Consequences

- **Breaking, with no migration path, by design.** Every existing `cobj` implementation must be
  rewritten: new signature, new contribution mechanism, and inverted sign convention (a reward is
  now a negative penalty). Every existing `ccnt` implementation keeps its signature. Because both
  fallbacks are now typed, a stale hook raises a `MethodError` on the first solve instead of
  silently dropping its term.
- **`reward_sign` is gone and `MaximumRatio` is no longer a special case** for custom objectives.
  The `get_k` homogenisation idiom is *not* affected and still applies: this removes the sign trap
  under a ratio objective, not the rescaling one.
- **The two extension points are now learn-once.** Both take
  `(model, <the thing you dispatch on>, optimiser, attrs)`; the objective hook adds `obj` ahead of
  the dispatch argument, and that is the only difference between them.
- `test_03c`'s NOC regression detector re-keys onto `isa(e.optimiser, NearOptimalCentering)`. This
  is strictly more precise than what it replaces: it names the estimator under test rather than
  detecting a type confusion that this ADR removes. Its `InertObjective` / `InertConstraint` cases
  invert from asserting a no-op to asserting a raise.

## Extends

- **[ADR 0008](0008-jump-model-assembly.md)** — the custom hooks are part of the assembly pipeline's
  public seam; this gives the objective hook the same fixed-arity, bundle-carrying shape the rest of
  the pipeline uses.
- **[ADR 0004](0004-typed-jump-model-state.md)** — the objective penalty accumulator (`:op`) is
  Model State reached through `add_to_objective_penalty!`, and is now the sole channel by which user
  terms enter the objective.
