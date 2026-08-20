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

## Amendment (2026-08-17)

Decision 4 was **stated but never implemented**, and the gap was found only when a later change
tripped over it. The expansion emitted `factory`, `factory_child` and `port_opt_view` as **bare
names**, and `AbstractPriorResult`, `sel`, `_ctx`, `_wprop` and `resolve_deferred_quantities`
with them. Every one of them is escaped into the caller, so every one of them resolved where the
struct is *declared*.

### How it surfaced

The contract check moved to the end of the module, so the expansion gained a call that records
the declaration in `PROPAGATABLE_CONTRACTS`. That call went in bare like the rest, and
`propagatable_register!` is private, so a declaration outside `PortfolioOptimisers` died at once:

```julia
UndefVarError: `propagatable_register!` not defined in `Main.WindowedEstimatorProbe`
```

The test suite's own probe module is such a declaration, so the whole of `test_08_moments`
errored rather than one assertion failing. A loud failure is what made the rest visible.

### What `factory` being exported does and does not buy

Nothing. An exported name arrives through `using` as an **implicit** binding, and a method
definition on an implicit binding does not extend it — Julia declares a **new** function of the
caller's own. Measured on 2026-08-17 with a module that wrote only `using PortfolioOptimisers`
and `using PortfolioOptimisers: @propagatable`:

- the declaration **compiled**, with no error and no warning;
- the contract **registered**, so the end-of-module check saw a healthy type;
- `NaiveUserProbe.factory !== PortfolioOptimisers.factory` — the caller got its own;
- `PortfolioOptimisers.factory(probe)` threw a **`MethodError`**.

So the promise in `@propagatable`'s docstring — that a type declared in another package slots
into PO's factory propagation chain — did not hold, and it failed **silently**. That is the
worst shape available: three of the four observable signals said the declaration was fine.

### The implementation

`POMOD = @__MODULE__` is taken in the macro body, where it is the module that **defines** the
macro. Each emitted name is built against it, e.g. `_factory = :($POMOD.factory)`. Interpolating
the module *object* rather than the name `PortfolioOptimisers` means the expansion needs **no
binding at all** in the caller, so a module that imports only `@propagatable` works.

`@propagatable` is not exported, contrary to decision 4's wording. The minimum a caller writes
is therefore `using PortfolioOptimisers: @propagatable`. Whether to export it is a separate,
public-API question and is not decided here.

### The rule for the next name added

Anything the expansion emits by name is looked up where the struct is *declared*. Qualify it
against `POMOD`. The test that catches a breach is `test/test_05_tools.jl`: it declares a type in
a module that imports **only** the macro, then asserts that the caller gained no `factory` of its
own and that `PortfolioOptimisers.factory` dispatches on the new type. A bare name fails that
test whether it is private (`UndefVarError`) or exported (silent shadow).
