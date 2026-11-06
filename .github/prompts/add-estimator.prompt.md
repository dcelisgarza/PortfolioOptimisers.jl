---
agent: ask
description: Step-by-step workflow for adding a new estimator to PortfolioOptimisers.jl.
---

Follow these steps to add a new estimator to PortfolioOptimisers.jl. Read the files that this prompt links before you start.

## Before you begin

This prompt carries the **order of the work**, and no rule of its own. Every rule it needs is in a standards file, and each step links to the section that owns it. Read these first:

- [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md) says which file owns each rule, and which check holds it.
- [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) owns the type roles, the constructors, validation, dispatch and exports.
- [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) is the Authority for every docstring section that the steps below name.
- [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md) says when to annotate a return type.
- Read a similar estimator file in `src/` as a reference.

## Step 1. Identify the correct abstract supertype

Decide which abstract type hierarchy this estimator belongs to:

- `AbstractEstimator` (base for all estimators)
- `AbstractCovarianceEstimator`, `AbstractExpectedReturnsEstimator`, `AbstractVarianceEstimator`, etc. (moments)
- `AbstractPriorEstimator` / `AbstractLowOrderPriorEstimator_A` / etc. (priors)
- `AbstractDenoiseEstimator`, `AbstractDetoneEstimator`, `AbstractPosdefEstimator` (matrix processing)
- Other domain-specific abstract type

If no suitable abstract type exists, define a new one. Its docstring follows [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Abstract types*, which owns the section list and its order.

An abstract type is **not exported** unless the maintainer says so. [`CLAUDE.md`](../../CLAUDE.md) § *Design rules* owns that rule, and `test/test_43_exported_abstract_type_census.jl` holds it.

## Step 2. Define the struct

Work in the source file of the subject, or in a new numbered file if the estimator is a separate component:

 1. Write `@concrete struct MyEstimator <: AbstractSupertype`, an inner constructor that validates its positional arguments and returns `new`, and an outer keyword constructor that carries the defaults. [`.github/instructions/julia-source-code.instructions.md`](../instructions/julia-source-code.instructions.md) § *Constructor Pattern* owns the shape, and § *Input Validation* owns the checks.
 2. Write the docstring as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Concrete struct types* states. A type that carries propagated, view or observation-weight parameters follows § *`@propagatable` concrete struct types* instead.
 3. Write one inline description per field, as § *Inline field docstrings* of the same file states.
 4. Read a real docstring rather than a copy of one. The § *Reference docstrings* table names one Unit per kind, and a Gate holds every Unit it names.

## Step 3. Implement the interface methods

Implement every method that the `# Interfaces` section of the abstract supertype requires. The common ones are these:

- `factory` and `port_opt_view`. Do not write them by hand for a type that propagates observation weights, a prior or a view. Declare it `@propagatable @concrete struct` and tag its fields. The macro, in `src/02_Tools/05_Propagatable.jl`, always makes `factory`, and it makes `port_opt_view` when a field carries the `@vprop` tag. Read `GeneralCovariance` in [`src/05_Moments/03_Covariance.jl`](../../src/05_Moments/03_Covariance.jl). Write a method by hand only for a rule the tags cannot express, as `factory(re::LinearModel, w::ObsWeights)::LinearModel` in [`src/05_Moments/20_Base_Regression.jl`](../../src/05_Moments/20_Base_Regression.jl) does.
- The domain-specific computation function (for example `Statistics.cov`, `prior`, `denoise!`).

Write a docstring for every method, as [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Section Structure for Functions* states.

## Step 4. Add return type annotations

Follow [`.github/instructions/julia-return-types.instructions.md`](../instructions/julia-return-types.instructions.md):

- Annotate a hand-written `factory` with `::MyEstimator`.
- Annotate validation helpers with `::Nothing`.
- Annotate passthrough methods with the abstract return type.

## Step 5. Add `arg_dict` / `field_dict` entries if needed

If a field or an argument has no entry yet in the dictionaries in [`src/01_Base/01_DocstringDictionaries/`](../../src/01_Base/01_DocstringDictionaries/), add the entry before you finish the docstring. [`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *Documentation Dictionaries* states which text a dictionary owns and when prose is permitted instead.

## Step 6. Export

Add the new type and any new public functions to the `export` statement at the bottom of the source file.

## Step 7. Add to API docs

Add the symbol under a suitable heading in the corresponding `docs/src/public_api/*.md` or `docs/src/private_api/*.md` file, on the side that its classification gives it:

````markdown
```@docs
MyEstimator
```
````

[`.github/instructions/julia-docstrings.instructions.md`](../instructions/julia-docstrings.instructions.md) § *`docs/src/public_api/` and `docs/src/private_api/` Markdown Files* owns the layout of that page, including when the page carries a bibliography block.

## Step 8. Write tests

Create or extend a `test/test_*.jl` file, as [`.github/instructions/julia-test-writing.instructions.md`](../instructions/julia-test-writing.instructions.md) states:

1. Test the validation of the constructor, every `@argcheck` condition.
2. Test normal use with valid inputs.
3. Test that `factory` propagates the weights.
4. Test that `port_opt_view` returns the correct type and slice.
5. Test each dispatch variant of the computation function.
6. Test the estimator composed with other estimators.

## Step 9. Wire the file into the sweep

If the addition creates a new file under `src/`, give it a row in [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) and open its sweep ticket in the same change. [`CLAUDE.md`](../../CLAUDE.md) § *Functionality you add* owns the four steps, and `test/test_45_sweep_census.jl` holds them.

## Step 10. Final checks

Run the pre-commit checks, the tests for the area you changed, and the doctests, in the order [`.github/prompts/pre-commit-and-test.prompt.md`](pre-commit-and-test.prompt.md) gives.
