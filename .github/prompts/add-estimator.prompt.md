---
agent: ask
description: Step-by-step workflow for adding a new estimator to PortfolioOptimisers.jl.
---

Follow these steps to add a new estimator to PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

Read the following to understand patterns and conventions:

- `.github/instructions/julia-source-code.instructions.md`
- `.github/instructions/julia-docstrings.instructions.md`
- `.github/instructions/julia-return-types.instructions.md`
- A similar existing estimator file in `src/` as a reference.

## Step 1 — Identify the correct abstract supertype

Decide which abstract type hierarchy this estimator belongs to:

- `AbstractEstimator` (base for all estimators)
- `AbstractCovarianceEstimator`, `AbstractExpectedReturnsEstimator`, `AbstractVarianceEstimator`, etc. (moments)
- `AbstractPriorEstimator` / `AbstractLowOrderPriorEstimator_A` / etc. (priors)
- `AbstractDenoiseEstimator`, `AbstractDetoneEstimator`, `AbstractPosdefEstimator` (matrix processing)
- Other domain-specific abstract type

If no suitable abstract type exists, define a new one (see `add-algorithm.prompt.md` for algorithm types and follow the same pattern for estimator abstract types).

## Step 2 — Define the struct

In the appropriate source file (or a new numbered file if this is a distinct component):

1. Write the docstring using `$(DocStringExtensions.TYPEDEF)` as the header.
2. Use `@concrete struct MyEstimator <: AbstractSupertype`.
3. Document all fields inline using `"$(field_dict[:key])"`.
4. Write the **inner constructor**: positional args, `@argcheck` / `assert_*` validation, `return new{typeof(f1), ...}(f1, ...)`.
5. Write the **outer constructor**: keyword args with default values, delegates to inner constructor.

````julia
"""
$(DocStringExtensions.TYPEDEF)

Description of what this estimator does.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyEstimator(;
        field1::Type1 = default1,
        field2::Type2 = default2
    ) -> MyEstimator

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:key])

# Examples

```jldoctest
julia> MyEstimator()
MyEstimator
  field1 ┴ default1
```

## Related

  - [`AbstractSupertype`](@ref)
  - [`my_function`](@ref)
"""
@concrete struct MyEstimator <: AbstractSupertype
    "$(field_dict[:key1])"
    field1
    "$(field_dict[:key2])"
    field2
    function MyEstimator(field1::Type1, field2::Type2)
        assert_nonempty_nonneg_finite_val(field1, :field1)
        return new{typeof(field1), typeof(field2)}(field1, field2)
    end
end
function MyEstimator(; field1::Type1 = default1, field2::Type2 = default2)
    return MyEstimator(field1, field2)
end
````

## Step 3 — Implement the interface methods

Implement all methods required by the abstract supertype's `# Interfaces` section. Common ones include:

- `factory(est::MyEstimator, w::ObsWeights)::MyEstimator` — returns a copy with observation weights propagated.
- `moment_view(est::MyEstimator, i)::MyEstimator` — returns a sliced view for windowed estimation.
- The domain-specific computation function (e.g., `Statistics.cov`, `prior`, `denoise!`).

Write a docstring for every method.

## Step 4 — Add return type annotations

Following `.github/instructions/julia-return-types.instructions.md`:

- Annotate `factory` with `::MyEstimator`.
- Annotate validation helpers with `::Nothing`.
- Annotate passthrough methods with the abstract return type.

## Step 5 — Add `arg_dict` / `field_dict` entries if needed

If any field or argument does not yet have an entry in the dictionaries in `src/01_Base.jl`, add them before finalising the docstring.

## Step 6 — Export

Add the new type and any new public functions to the `export` statement at the bottom of the source file.

## Step 7 — Add to API docs

Add the symbol to the corresponding `docs/src/api/*.md` file under an appropriate heading:

````markdown
```@docs
MyEstimator
```
````

## Step 8 — Write tests

Create or extend a `test/test-*.jl` file following `.github/instructions/julia-test-writing.instructions.md`:

1. Test constructor validation (all `@argcheck` conditions).
2. Test normal usage with valid inputs.
3. Test `factory` propagates weights correctly.
4. Test `moment_view` returns the correct type and slice.
5. Test each dispatch variant of the computation function.
6. Test composability with other estimators.

## Step 9 — Final checks

Run the full pre-commit, test, and doctest suite following `.github/prompts/pre-commit-and-test.prompt.md`.

All three steps must pass before committing.
