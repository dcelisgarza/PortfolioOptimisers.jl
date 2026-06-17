---
status: accepted
---

# `@propagatable` macro design

## Context

ADR 0001 adopted `@propagatable` as the mechanism for auto-generating pure-propagation
`factory` methods. This ADR records the implementation decisions made when turning
the spike into production code.

## Decisions

### 1. Explicit opt-in field tagging with `@prop`

**Decision:** fields that participate in `factory` propagation must be tagged with `@prop`
inside the struct body. Untagged fields pass through unchanged regardless of their type.

**Why:** runtime type dispatch alone (`factory_child` dispatching on
`Union{<:AbstractEstimator,<:AbstractAlgorithm}`) is insufficient. A field may hold
an eligible-typed value that should not be recursed — for example, equilibrium portfolio
weights stored as an estimator, or a configuration-only sub-estimator. The domain
decision about which fields carry runtime data cannot be recovered from types alone; it
must be stated explicitly at the definition site.

**Consequence:** opt-in is the safer default. A missed `@prop` tag causes a field to be
skipped (the old identity passthrough remains correct); an incorrect `@prop` tag would
cause `factory` to recurse into an inert field (potentially a silent bug). New fields
default to inert.

### 2. Composition order: `@propagatable` outermost

**Decision:** `@propagatable @concrete struct Foo ...` — `@propagatable` is the outermost macro.

**Why:** in Julia, `@outer @inner expr` means `@outer` receives the unevaluated
`@inner` AST node. `@propagatable` must see the raw struct body (including `@prop` tags)
before `@concrete` rewrites it. The reverse — `@concrete @propagatable struct` — would
require `@concrete` to understand `@propagatable`, which it cannot.

`@propagatable` recursively unwraps arbitrary `:macrocall` chains until it finds the
`:struct` node, processes `@prop` tags, and re-emits the full original chain (cleaned up)
plus the factory method. This makes `@propagatable` compose correctly with any future
macro layered between it and `struct`.

### 3. Docstring forwarding via `Base.@__doc__`

**Decision:** the macro expansion uses `Base.@__doc__ $chain` as its first emitted
expression.

**Why:** without `Base.@__doc__`, a docstring placed before `@propagatable @concrete struct
Foo ...` is consumed by Julia's doc system but not forwarded to `Foo` — it is silently
dropped. `Base.@__doc__` is the standard Julia pattern for macros that need to be
docstring-transparent; it forwards any preceding docstring through the `@concrete`
expansion to `Foo`.

### 4. Qualified `PortfolioOptimisers.factory` and `factory_child` in the expansion

**Decision:** the generated factory method is `PortfolioOptimisers.factory(x::Foo, ...)`
and the per-field helper call is `PortfolioOptimisers.factory_child(...)`, both fully
qualified.

**Why:** `@propagatable` is exported for external use — a user in another package can write
`@propagatable @concrete struct MyEstimator <: AbstractEstimator ...` and their type will
slot into PO's factory propagation chain. Unqualified names would add `factory` and
`factory_child` to the *user's* module, not to `PortfolioOptimisers`.

### 5. `@prop` error stub

**Decision:** a `macro c(expr)` that `error`s with a clear message is defined alongside
`@propagatable`.

**Why:** without the stub, `@prop field` outside a `@propagatable` body produces Julia's
generic "macro not found" error. The stub gives a diagnostic pointing back to the
intended usage.

### 6. Source layout

- `factory_child` helpers: `src/02_Tools.jl`, immediately after the existing `factory`
  fallbacks. Lives near the `factory` definitions it supports.
- `@propagatable`, `@prop`, helpers.
- The `AbstractCovarianceEstimator` extension of `factory_child` (needed for the full
  migration) lives in `src/08_Moments/01_Base_Moments.jl` after
  `AbstractCovarianceEstimator` is defined — `02_Tools.jl` cannot reference it because
  it is included before the `08_Moments` files.

## AST shape reference

Julia's parser fuses docstrings with their target:

```julia
"doc" \n @prop a   →   Core.@doc "doc" @prop(a)
```

as a single `:macrocall` node with:

- `args[1]`: `GlobalRef(Core, :@doc)`
- `args[2]`: `LineNumberNode`
- `args[3]`: `"doc"` (String)
- `args[4]`: `Expr(:macrocall, Symbol("@prop"), LineNumberNode, :a)`

`@propagatable` strips `@prop` by replacing `args[4]` with the bare field expression
(`:a`), preserving the `Core.@doc` wrapper so field-level docstrings survive.

## Verification

- `factory(ex, w)` propagates `ObsWeights` into `@prop`-tagged `inner`,
  leaves untagged `config` unchanged ✓
- `@inferred factory(ex, w)` — type-stable ✓
