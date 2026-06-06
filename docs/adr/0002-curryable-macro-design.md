---
status: accepted
---

# `@curryable` macro design

## Context

ADR 0001 adopted `@curryable` as the mechanism for auto-generating pure-propagation
`factory` methods. This ADR records the implementation decisions made when turning
the spike into production code.

## Decisions

### 1. Explicit opt-in field tagging with `@c`

**Decision:** fields that participate in `factory` propagation must be tagged with `@c`
inside the struct body. Untagged fields pass through unchanged regardless of their type.

**Why:** runtime type dispatch alone (`_factory_child` dispatching on
`Union{<:AbstractEstimator,<:AbstractAlgorithm}`) is insufficient. A field may hold
an eligible-typed value that should not be recursed — for example, equilibrium portfolio
weights stored as an estimator, or a configuration-only sub-estimator. The domain
decision about which fields carry runtime data cannot be recovered from types alone; it
must be stated explicitly at the definition site.

**Consequence:** opt-in is the safer default. A missed `@c` tag causes a field to be
skipped (the old identity passthrough remains correct); an incorrect `@c` tag would
cause `factory` to recurse into an inert field (potentially a silent bug). New fields
default to inert.

### 2. Composition order: `@curryable` outermost

**Decision:** `@curryable @concrete struct Foo ...` — `@curryable` is the outermost macro.

**Why:** in Julia, `@outer @inner expr` means `@outer` receives the unevaluated
`@inner` AST node. `@curryable` must see the raw struct body (including `@c` tags)
before `@concrete` rewrites it. The reverse — `@concrete @curryable struct` — would
require `@concrete` to understand `@curryable`, which it cannot.

`@curryable` recursively unwraps arbitrary `:macrocall` chains until it finds the
`:struct` node, processes `@c` tags, and re-emits the full original chain (cleaned up)
plus the factory method. This makes `@curryable` compose correctly with any future
macro layered between it and `struct`.

### 3. Docstring forwarding via `Base.@__doc__`

**Decision:** the macro expansion uses `Base.@__doc__ $chain` as its first emitted
expression.

**Why:** without `Base.@__doc__`, a docstring placed before `@curryable @concrete struct
Foo ...` is consumed by Julia's doc system but not forwarded to `Foo` — it is silently
dropped. `Base.@__doc__` is the standard Julia pattern for macros that need to be
docstring-transparent; it forwards any preceding docstring through the `@concrete`
expansion to `Foo`.

### 4. Qualified `PortfolioOptimisers.factory` and `_factory_child` in the expansion

**Decision:** the generated factory method is `PortfolioOptimisers.factory(x::Foo, ...)`
and the per-field helper call is `PortfolioOptimisers._factory_child(...)`, both fully
qualified.

**Why:** `@curryable` is exported for external use — a user in another package can write
`@curryable @concrete struct MyEstimator <: AbstractEstimator ...` and their type will
slot into PO's factory propagation chain. Unqualified names would add `factory` and
`_factory_child` to the *user's* module, not to `PortfolioOptimisers`.

### 5. `@c` error stub

**Decision:** a `macro c(expr)` that `error`s with a clear message is defined alongside
`@curryable`.

**Why:** without the stub, `@c field` outside a `@curryable` body produces Julia's
generic "macro not found" error. The stub gives a diagnostic pointing back to the
intended usage.

### 6. Source layout

- `_factory_child` helpers: `src/02_Tools.jl`, immediately after the existing `factory`
  fallbacks. Lives near the `factory` definitions it supports.
- `@curryable`, `@c`, helpers, and the `_CurryableExample` dummy type:
  `src/02b_Curryable.jl`, included after `02_Tools.jl`.
- The `AbstractCovarianceEstimator` extension of `_factory_child` (needed for the full
  migration) lives in `src/08_Moments/01_Base_Moments.jl` after
  `AbstractCovarianceEstimator` is defined — `02_Tools.jl` cannot reference it because
  it is included before the `08_Moments` files.

## AST shape reference

Julia's parser fuses docstrings with their target:

```julia
"doc" \n @c a   →   Core.@doc "doc" @c(a)
```

as a single `:macrocall` node with:

- `args[1]`: `GlobalRef(Core, :@doc)`
- `args[2]`: `LineNumberNode`
- `args[3]`: `"doc"` (String)
- `args[4]`: `Expr(:macrocall, Symbol("@c"), LineNumberNode, :a)`

`@curryable` strips `@c` by replacing `args[4]` with the bare field expression
(`:a`), preserving the `Core.@doc` wrapper so field-level docstrings survive.

## Verification

Implemented and validated in `src/02b_Curryable.jl`:

- `_CurryableExample()` constructs with defaults ✓
- `factory(ex, w)` propagates `ObsWeights` into `@c`-tagged `inner`,
  leaves untagged `config` unchanged ✓
- `@inferred factory(ex, w)` — type-stable ✓
- `@doc _CurryableExample` — struct-level and field-level docstrings render ✓
