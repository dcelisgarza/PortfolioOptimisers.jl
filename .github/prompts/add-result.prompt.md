---
agent: ask
description: Step-by-step workflow for adding a new result type to PortfolioOptimisers.jl.
---

Follow these steps to add a new result type to PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

Read the following to understand patterns and conventions:

- `.github/instructions/julia-source-code.instructions.md`
- `.github/instructions/julia-docstrings.instructions.md`
- `.github/instructions/julia-return-types.instructions.md`
- A similar existing result type (e.g., `LowOrderPrior` in `src/13_Prior/`) as a reference.

## Key rules for result types

- Results are returned by functions that consume estimators.
- Results exist only when the returned data is complex and can itself be used as input to further computation.
- Results can be passed back to the same function (passthrough/no-op pattern): `f(::AbstractXResult, args...) = result`.
- Results must never be required as arguments in user-facing API signatures — they are an optional shortcut.

## Step 1 — Identify the correct abstract supertype

Decide which abstract result type this result belongs to:

- `AbstractResult` (base for all results)
- `AbstractPriorResult`, `AbstractLowOrderPriorResult`, etc.
- Domain-specific abstract result type

If no suitable abstract type exists, define a new one before defining the concrete result.

### Defining a new abstract result type

```julia
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all my-domain result types.

# Related

  - [`AbstractResult`](@ref)
  - [`my_function`](@ref)
"""
abstract type AbstractMyResult <: AbstractResult end
```

## Step 2 — Define the result struct

````julia
"""
$(DocStringExtensions.TYPEDEF)

Result returned by [`my_function`](@ref) when called with a `MyEstimator`.

# Fields

$(DocStringExtensions.FIELDS)

# Examples

```jldoctest
julia> r = my_function(MyEstimator(), X)
MyResult
  field1 ┴ ...
  field2 ┴ ...
```

## Related

- [`AbstractMyResult`](@ref)
- [`my_function`](@ref)
- [`MyEstimator`](@ref)
"""
@concrete struct MyResult <: AbstractMyResult
    "$(field_dict[:key1])"
    field1
    "$(field_dict[:key2])"
    field2
end
````

Result types typically do not need an inner constructor with validation unless the struct could be constructed independently outside the producing function. Most validation is done in the producing estimator/function.

## Step 3 — Implement the passthrough method

Add a passthrough method to the consuming function so that a result can be re-used directly:

```julia
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Passes `result` through unchanged.

Allows a pre-computed `MyResult` to be used wherever a `MyEstimator` is expected.

# Arguments

  - `result::AbstractMyResult`: Pre-computed result.

# Returns

  - $(ret_dict[:key])

# Related

  - [`AbstractMyResult`](@ref)
  - [`my_function`](@ref)
"""
function my_function(result::AbstractMyResult, args...; kwargs...)::AbstractMyResult
    return result
end
```

## Step 4 — Update the producing function's return type annotation

If `my_function(est::MyEstimator, ...)` is not yet annotated:

```julia
function my_function(est::MyEstimator, X::MatNum; kwargs...)::MyResult
    ...
    return MyResult(field1, field2)
end
```

## Step 5 — Add `field_dict` entries if needed

Add any missing entries to the dictionaries in `src/01_Base.jl`.

## Step 6 — Export

Add the new result type to the `export` statement at the bottom of the source file.

## Step 7 — Add to API docs

Add the symbol to the corresponding `docs/src/api/*.md` file:

````markdown
```@docs
MyResult
```
````

## Step 8 — Write tests

Create or extend a `test/test-*.jl` file following `.github/instructions/julia-test-writing.instructions.md`:

1. Test that the result is returned correctly by the producing function.
2. Test the passthrough method: `my_function(result, ...) === result`.
3. Test that all expected fields are present and correctly typed.

## Step 9 — Final checks

Run the full pre-commit, test, and doctest suite following `.github/prompts/pre-commit-and-test.prompt.md`.

All three steps must pass before committing.
