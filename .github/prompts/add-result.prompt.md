---
agent: ask
description: Step-by-step workflow for adding a new result type to PortfolioOptimisers.jl.
---

Follow these steps to add a new result type to PortfolioOptimisers.jl. Read the files that this prompt links before you start.

## Before you begin

This prompt carries the **order of the work**, and no rule of its own. Every rule it needs is in a standards file, and each step links to the section that owns it. Read these first:

- [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md) says which file owns each rule, and which check holds it.
- [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) owns the type roles, the constructors, validation, dispatch and exports.
- [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) is the Authority for every docstring section that the steps below name.
- [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md) says when to annotate a return type.
- Read `LowOrderPrior` in [`src/10_Prior/01_Base_Prior.jl`](../../src/10_Prior/01_Base_Prior.jl) as a reference result type.

## Key rules for result types

[`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Estimator, Algorithm, and Result Roles* owns the three roles. Four points decide whether a result type is the right choice:

- A function that consumes an estimator returns a result.
- A result exists only when the returned data is complex and can itself be the input of a further computation.
- A caller can pass a result back to the same function, which returns it unchanged, the passthrough or no-op pattern: `f(::AbstractXResult, args...) = result`.
- A user-facing function must never require a result as an argument. A result is an optional shortcut.

An estimator never holds a Result in a field. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and the field's type bound enforces it.

## Step 1. Identify the correct abstract supertype

Decide which abstract result type this result belongs to:

- `AbstractResult` (base for all results)
- `AbstractPriorResult`, `AbstractClusteringResult`, etc.
- Domain-specific abstract result type

If no suitable abstract type exists, define a new one before defining the concrete result. Its docstring follows [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Abstract types*.

An abstract type is **not exported** unless the maintainer says so. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and `test/test_43_exported_abstract_type_census.jl` holds it.

## Step 2. Define the result struct

Write `@concrete struct MyResult <: AbstractMyResult` and document it as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Concrete struct types* states. Each field carries an inline description, as § *Inline field docstrings* states.

A result type usually needs no inner constructor with validation. The producing estimator or function validates instead. Write one only when a caller can construct the struct outside the producing function. [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Constructor Pattern* owns the shape when you do.

## Step 3. Implement passthrough and interface methods

Add the passthrough method, so that a caller can pass a precomputed result in place of an estimator:

```julia
function my_function(result::AbstractMyResult, args...; kwargs...)::AbstractMyResult
    return result
end
```

Document it as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Section Structure for Functions* states.

A Result that a view or a propagation reaches also needs `port_opt_view` or `factory`. Do not write them by hand when field tags can express the cut. Declare it `@propagatable @concrete struct` and tag its fields, as `Turnover` in [`src/12_Turnover.jl`](../../src/12_Turnover.jl) does. The macro, in `src/02_Tools/05_Propagatable.jl`, always makes `factory`, and it makes `port_opt_view` when a field carries the `@vprop` tag. The docstring then follows [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *`@propagatable` concrete struct types*. Write a method by hand only for a cut the tags cannot express, as `port_opt_view(pr::LowOrderPrior, i, args...)::LowOrderPrior` in [`src/10_Prior/01_Base_Prior.jl`](../../src/10_Prior/01_Base_Prior.jl) does.

## Step 4. Update the producing function's return type annotation

Follow [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md). Annotate `my_function(est::MyEstimator, ...)` with `::MyResult` when it always returns that concrete type.

## Step 5. Add `*_dict` entries if needed

Add any missing entries to the dictionaries in [`src/01_Base/01_DocstringDictionaries/`](../../src/01_Base/01_DocstringDictionaries/). [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Documentation Dictionaries* states which text a dictionary owns and when prose is permitted instead.

## Step 6. Export

Add the new result type to the `export` statement at the bottom of the source file.

## Step 7. Add to API docs

Add the symbol to the corresponding `docs/src/public_api/*.md` or `docs/src/private_api/*.md` file, on the side that its classification gives it:

````markdown
```@docs
MyResult
```
````

[`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *`docs/src/public_api/` and `docs/src/private_api/` Markdown Files* owns the layout of that page, including when the page carries a bibliography block.

## Step 8. Write tests

Create or extend a `test/test_*.jl` file, as [`.github/instructions/julia-test-writing.instructions.md`](../instructions/julia-test-writing.instructions.md) states:

1. Test that the producing function returns the correct result.
2. Test the passthrough method: `my_function(result, ...) === result`.
3. Test that every expected field is present and has the correct type.

## Step 9. Wire the file into the sweep

If the addition creates a new file under `src/`, give it a row in [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) and open its sweep ticket in the same change. [`CLAUDE.md`](../../CLAUDE.md) § *Functionality you add* owns the four steps, and `test/test_45_sweep_census.jl` holds them.

## Step 10. Final checks

Run the pre-commit checks, the tests for the area you changed, and the doctests, in the order [`.github/prompts/pre-commit-and-test.prompt.md`](pre-commit-and-test.prompt.md) gives.
