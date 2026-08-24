---
agent: ask
description: Step-by-step workflow for adding a new result type to PortfolioOptimisers.jl.
---

Follow these steps to add a new result type to PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

This prompt carries the **order of the work**, and no rule of its own. Every rule it needs lives in a standards file, and each step links to the section that owns it. Read these first:

- [`STANDARDS.md`](../../STANDARDS.md) — which file owns which rule, and which check holds it.
- [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) — type roles, constructors, validation, dispatch, exports.
- [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) — the Authority for every docstring section named below.
- [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md) — when to annotate a return type.
- `LowOrderPrior` in [`src/13_Prior/01_Base_Prior.jl`](../../src/13_Prior/01_Base_Prior.jl) as a reference result type.

## Key rules for result types

[`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Estimator, Algorithm, and Result Roles* owns the three roles. The four points that decide whether a result type is the right answer:

- Results are returned by functions that consume estimators.
- Results exist only when the returned data is complex and can itself be used as input to further computation.
- Results can be passed back to the same function (passthrough/no-op pattern): `f(::AbstractXResult, args...) = result`.
- Results must never be required as arguments in user-facing API signatures — they are an optional shortcut.

An estimator never holds a Result in a field. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and the field's type bound enforces it.

## Step 1 — Identify the correct abstract supertype

Decide which abstract result type this result belongs to:

- `AbstractResult` (base for all results)
- `AbstractPriorResult`, `AbstractClusteringResult`, etc.
- Domain-specific abstract result type

If no suitable abstract type exists, define a new one before defining the concrete result. Its docstring follows [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Abstract types*.

An abstract type is **not exported** unless the maintainer says so. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and `test/test_43_exported_abstract_type_census.jl` holds it.

## Step 2 — Define the result struct

Write `@concrete struct MyResult <: AbstractMyResult` and document it as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Concrete struct types* states. Each field carries an inline description, as § *Inline field docstrings* states.

A result type usually needs no inner constructor with validation. The producing estimator or function validates instead. Write one only when a caller can construct the struct outside the producing function. [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Constructor Pattern* owns the shape when you do.

## Step 3 — Implement passthrough and interface methods

Add the passthrough method that lets a pre-computed result stand where an estimator is expected:

```julia
function my_function(result::AbstractMyResult, args...; kwargs...)::AbstractMyResult
    return result
end
```

Document it as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Section Structure for Functions* states.

Common interface methods to implement include:

- `factory(res::MyResult, w::ObsWeights)::MyResult` — returns a copy with observation weights propagated.
- `port_opt_view(res::MyResult, i)::MyResult` — returns a sliced view.

## Step 4 — Update the producing function's return type annotation

Follow [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md). Annotate `my_function(est::MyEstimator, ...)` with `::MyResult` when it always returns that concrete type.

## Step 5 — Add `*_dict` entries if needed

Add any missing entries to the dictionaries in [`src/01_Base.jl`](../../src/01_Base.jl). [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Documentation Dictionaries* states which text a dictionary owns and when prose is permitted instead.

## Step 6 — Export

Add the new result type to the `export` statement at the bottom of the source file.

## Step 7 — Add to API docs

Add the symbol to the corresponding `docs/src/api/*.md` file:

````markdown
```@docs
MyResult
```
````

[`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *`docs/src/api/` Markdown Files* owns the layout of that page, including when the page carries a bibliography block.

## Step 8 — Write tests

Create or extend a `test/test_*.jl` file following [`.github/instructions/julia-test-writing.instructions.md`](../instructions/julia-test-writing.instructions.md):

1. Test that the result is returned correctly by the producing function.
2. Test the passthrough method: `my_function(result, ...) === result`.
3. Test that all expected fields are present and correctly typed.

## Step 9 — Wire the file into the sweep

If the addition creates a new file under `src/`, give it a row in [`sweep/manifest.toml`](../../sweep/manifest.toml) and open its sweep ticket in the same change. [`CLAUDE.md`](../../CLAUDE.md) § *Functionality you add* owns the four steps, and `test/test_45_sweep_census.jl` holds them.

## Step 10 — Final checks

Run the full pre-commit, test, and doctest suite following [`.github/prompts/pre-commit-and-test.prompt.md`](pre-commit-and-test.prompt.md).

All three steps must pass before committing.
