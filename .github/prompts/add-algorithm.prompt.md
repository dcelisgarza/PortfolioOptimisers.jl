---
agent: ask
description: Step-by-step workflow for adding a new algorithm to PortfolioOptimisers.jl.
---

Follow these steps to add a new algorithm to PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

This prompt carries the **order of the work**, and no rule of its own. Every rule it needs lives in a standards file, and each step links to the section that owns it. Read these first:

- [`STANDARDS.md`](../../STANDARDS.md) — which file owns which rule, and which check holds it.
- [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) — type roles, constructors, validation, dispatch, exports.
- [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) — the Authority for every docstring section named below.
- [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md) — when to annotate a return type.
- A similar existing algorithm definition in `src/` as a reference.

## Key rule: algorithms are not user-facing

Algorithms are internal dispatch mechanisms. They must never appear in high-level, user-facing API signatures as required arguments. They always live as a field inside an estimator. [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Estimator, Algorithm, and Result Roles* owns the three roles.

## Step 1 — Identify the correct abstract supertype

Decide which abstract algorithm type this algorithm belongs to:

- `AbstractAlgorithm` (base for all algorithms)
- `AbstractMomentAlgorithm` (`FullMoment`, `SemiMoment`)
- `AbstractDenoiseAlgorithm`, `AbstractMatrixProcessingAlgorithm`
- `AbstractDistanceAlgorithm`
- Other domain-specific abstract algorithm type

If no suitable abstract type exists, define a new abstract algorithm type first. Its docstring follows [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Abstract types*, which owns the section list and its order.

An abstract type is **not exported** unless the maintainer says so. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and `test/test_43_exported_abstract_type_census.jl` holds it.

Read `AbstractDenoiseAlgorithm` in [`src/05_Denoise.jl`](../../src/05_Denoise.jl) for a real abstract-type docstring. The § *Reference docstrings* table names it, and a Gate holds it.

## Step 2 — Define the algorithm struct

An algorithm is often a parameterless selector tag, and sometimes it carries parameters.

- A **selector tag** is a bare `struct MyAlgorithm <: AbstractMyAlgorithm end`. Read `SpectralDenoise` in [`src/05_Denoise.jl`](../../src/05_Denoise.jl) for its docstring.
- A **parameterised algorithm** is a `@concrete struct` with an inner constructor that validates and an outer keyword constructor that carries the defaults. [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Constructor Pattern* owns the shape, and § *Input Validation* owns the checks. Read `ShrunkDenoise` in [`src/05_Denoise.jl`](../../src/05_Denoise.jl) for its docstring.

Both kinds are documented as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Concrete struct types* states, and each field carries an inline description as § *Inline field docstrings* states.

## Step 3 — Implement the dispatch and interface methods

Write the method that the estimator calls internally when it holds this algorithm, and document it as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Section Structure for Functions* states. Read `denoise!` and `_denoise!` in [`src/05_Denoise.jl`](../../src/05_Denoise.jl) for a public and a private function docstring.

Common interface methods to implement include:

- `factory(alg::MyAlgorithm, w::ObsWeights)::MyAlgorithm` — returns a copy with observation weights propagated.
- `port_opt_view(alg::MyAlgorithm, i)::MyAlgorithm` — returns a sliced view.

## Step 4 — Add return type annotations

Follow [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md). Annotate the dispatch method with `::ReturnType` when it always returns the same concrete type.

## Step 5 — Add `*_dict` entries if needed

Add any missing entries to the dictionaries in [`src/01_Base/01_DocstringDictionaries.jl`](../../src/01_Base/01_DocstringDictionaries.jl). [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Documentation Dictionaries* states which text a dictionary owns and when prose is permitted instead.

## Step 6 — Export

Add the new algorithm type and any new public functions to the `export` statement at the bottom of the source file.

## Step 7 — Add to API docs

Add the symbol to the corresponding `docs/src/api/*.md` file:

````markdown
```@docs
MyAlgorithm
```
````

[`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *`docs/src/api/` Markdown Files* owns the layout of that page, including when the page carries a bibliography block.

## Step 8 — Write tests

Create or extend a `test/test_*.jl` file following [`.github/instructions/julia-test-writing.instructions.md`](../instructions/julia-test-writing.instructions.md):

1. If the algorithm has parameters, test constructor validation.
2. Test the dispatch method with valid inputs.
3. Test that the algorithm integrates correctly when composed inside its target estimator.

## Step 9 — Wire the file into the sweep

If the addition creates a new file under `src/`, give it a row in [`sweep/manifest.toml`](../../sweep/manifest.toml) and open its sweep ticket in the same change. [`CLAUDE.md`](../../CLAUDE.md) § *Functionality you add* owns the four steps, and `test/test_45_sweep_census.jl` holds them.

## Step 10 — Final checks

Run the full pre-commit, test, and doctest suite following [`.github/prompts/pre-commit-and-test.prompt.md`](pre-commit-and-test.prompt.md).

All three steps must pass before committing.
