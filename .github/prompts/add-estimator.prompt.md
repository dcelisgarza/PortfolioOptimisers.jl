---
agent: ask
description: Step-by-step workflow for adding a new estimator to PortfolioOptimisers.jl.
---

Follow these steps to add a new estimator to PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

This prompt carries the **order of the work**, and no rule of its own. Every rule it needs lives in a standards file, and each step links to the section that owns it. Read these first:

- [`STANDARDS.md`](../../STANDARDS.md) — which file owns which rule, and which check holds it.
- [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) — type roles, constructors, validation, dispatch, exports.
- [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) — the Authority for every docstring section named below.
- [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md) — when to annotate a return type.
- A similar existing estimator file in `src/` as a reference.

## Step 1 — Identify the correct abstract supertype

Decide which abstract type hierarchy this estimator belongs to:

- `AbstractEstimator` (base for all estimators)
- `AbstractCovarianceEstimator`, `AbstractExpectedReturnsEstimator`, `AbstractVarianceEstimator`, etc. (moments)
- `AbstractPriorEstimator` / `AbstractLowOrderPriorEstimator_A` / etc. (priors)
- `AbstractDenoiseEstimator`, `AbstractDetoneEstimator`, `AbstractPosdefEstimator` (matrix processing)
- Other domain-specific abstract type

If no suitable abstract type exists, define a new one. Its docstring follows [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Abstract types*, which owns the section list and its order.

An abstract type is **not exported** unless the maintainer says so. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and `test/test_43_exported_abstract_type_census.jl` holds it.

## Step 2 — Define the struct

In the appropriate source file (or a new numbered file if this is a distinct component):

 1. Write `@concrete struct MyEstimator <: AbstractSupertype`, an inner constructor that validates its positional arguments and returns `new`, and an outer keyword constructor that carries the defaults. [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Constructor Pattern* owns the shape, and § *Input Validation* owns the checks.
 2. Write the docstring as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Concrete struct types* states. A type that carries propagated, view or observation-weight parameters follows § *`@propagatable` concrete struct types* instead.
 3. Write one inline description per field, as § *Inline field docstrings* of the same file states.
 4. Read a real docstring rather than a copy of one. The § *Reference docstrings* table names one Unit per kind, and a Gate holds every Unit it names.

## Step 3 — Implement the interface methods

Implement all methods required by the abstract supertype's `# Interfaces` section. Common ones include:

- `factory(est::MyEstimator, w::ObsWeights)::MyEstimator` — returns a copy with observation weights propagated.
- `port_opt_view(est::MyEstimator, i)::MyEstimator` — returns a sliced view.
- The domain-specific computation function (for example `Statistics.cov`, `prior`, `denoise!`).

Write a docstring for every method, as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Section Structure for Functions* states.

## Step 4 — Add return type annotations

Follow [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md):

- Annotate `factory` with `::MyEstimator`.
- Annotate validation helpers with `::Nothing`.
- Annotate passthrough methods with the abstract return type.

## Step 5 — Add `arg_dict` / `field_dict` entries if needed

If any field or argument does not yet have an entry in the dictionaries in [`src/01_Base.jl`](../../src/01_Base.jl), add them before finalising the docstring. [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Documentation Dictionaries* states which text a dictionary owns and when prose is permitted instead.

## Step 6 — Export

Add the new type and any new public functions to the `export` statement at the bottom of the source file.

## Step 7 — Add to API docs

Add the symbol to the corresponding `docs/src/api/*.md` file under an appropriate heading:

````markdown
```@docs
MyEstimator
```
````

[`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *`docs/src/api/` Markdown Files* owns the layout of that page, including when the page carries a bibliography block.

## Step 8 — Write tests

Create or extend a `test/test_*.jl` file following [`.github/instructions/julia-test-writing.instructions.md`](../instructions/julia-test-writing.instructions.md):

1. Test constructor validation (all `@argcheck` conditions).
2. Test normal usage with valid inputs.
3. Test `factory` propagates weights correctly.
4. Test `port_opt_view` returns the correct type and slice.
5. Test each dispatch variant of the computation function.
6. Test composability with other estimators.

## Step 9 — Wire the file into the sweep

If the addition creates a new file under `src/`, give it a row in [`sweep/manifest.toml`](../../sweep/manifest.toml) and open its sweep ticket in the same change. [`CLAUDE.md`](../../CLAUDE.md) § *Functionality you add* owns the four steps, and `test/test_45_sweep_census.jl` holds them.

## Step 10 — Final checks

Run the full pre-commit, test, and doctest suite following [`.github/prompts/pre-commit-and-test.prompt.md`](pre-commit-and-test.prompt.md).

All three steps must pass before committing.
