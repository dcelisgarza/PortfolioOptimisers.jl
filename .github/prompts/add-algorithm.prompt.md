---
agent: ask
description: Step-by-step workflow for adding a new algorithm to PortfolioOptimisers.jl.
---

Follow these steps to add a new algorithm to PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

Read the following to understand patterns and conventions:

- `.github/instructions/julia-source-code.instructions.md`
- `.github/instructions/API-docstrings.instructions.md`
- `.github/instructions/julia-return-types.instructions.md`
- A similar existing algorithm definition in `src/` as a reference.

## Key rule: algorithms are not user-facing

Algorithms are internal dispatch mechanisms. They must never appear in high-level, user-facing API signatures as required arguments. They always live as a field inside an estimator.

## Step 1 — Identify the correct abstract supertype

Decide which abstract algorithm type this algorithm belongs to:

- `AbstractAlgorithm` (base for all algorithms)
- `AbstractMomentAlgorithm` (`Full`, `Semi`)
- `AbstractDenoiseAlgorithm`, `AbstractDetoneAlgorithm`
- `AbstractPosdefAlgorithm`
- Other domain-specific abstract algorithm type

If no suitable abstract type exists, define a new abstract algorithm type first.

### Defining a new abstract algorithm type

````julia
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all my-domain algorithm types.

All concrete algorithm types for my domain should subtype `AbstractMyAlgorithm`.

# Interfaces

  - `my_algorithm_function(alg::AbstractMyAlgorithm, arg::Type) -> ReturnType`: What it does.

### Arguments

  - `alg`: Concrete subtype.
  - `arg`: Description.

### Returns

  - `result::ReturnType`: Description.

### Examples

```jldoctest
julia> struct MyNewAlgorithm <: PortfolioOptimisers.AbstractMyAlgorithm end
...
```

## Related

- [`AbstractAlgorithm`](@ref)
- [`my_algorithm_function`](@ref)
"""
abstract type AbstractMyAlgorithm <: AbstractAlgorithm end
````

## Step 2 — Define the algorithm struct

Algorithms are often parameterless singletons:

```julia
"""
$(DocStringExtensions.TYPEDEF)

Implements my specific algorithm variant.

# Related

  - [`AbstractMyAlgorithm`](@ref)
  - [`my_algorithm_function`](@ref)
"""
struct MyAlgorithm <: AbstractMyAlgorithm end
```

If the algorithm has parameters:

````julia
"""
$(DocStringExtensions.TYPEDEF)

Implements my parameterised algorithm variant.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyAlgorithm(;
        param::Real = 0.5
    ) -> MyAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - `0 < param < 1`.

# Examples

```jldoctest
julia> MyAlgorithm()
MyAlgorithm
  param ┴ Float64: 0.5
```

## Related

- [`AbstractMyAlgorithm`](@ref)
"""
@concrete struct MyAlgorithm <: AbstractMyAlgorithm
    "$(field_dict[:key])"
    param
    function MyAlgorithm(param::Real)
        @argcheck 0 < param < 1 DomainError(param, "param must be in (0, 1)")
        return new{typeof(param)}(param)
    end
end
function MyAlgorithm(; param::Real = 0.5)
    return MyAlgorithm(param)
end
````

## Step 3 — Implement the dispatch method

Write the method that the estimator calls internally when it holds this algorithm:

```julia
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the my-domain computation using `MyAlgorithm`.

# Arguments

  - `alg::MyAlgorithm`: The algorithm.
  - `arg::Type`: Description.

# Returns

  - $(ret_dict[:key])

# Related

  - [`AbstractMyAlgorithm`](@ref)
  - [`MyAlgorithm`](@ref)
"""
function my_algorithm_function(::MyAlgorithm, arg::Type)::ReturnType
    ...
end
```

## Step 4 — Add return type annotations

- Annotate the dispatch method with `::ReturnType` if it always returns the same concrete type.

## Step 5 — Add `arg_dict` / `field_dict` entries if needed

Add any missing entries to the dictionaries in `src/01_Base.jl`.

## Step 6 — Export

Add the new algorithm type and any new public functions to the `export` statement at the bottom of the source file.

## Step 7 — Add to API docs

Add the symbol to the corresponding `docs/src/api/*.md` file:

````markdown
```@docs
MyAlgorithm
```
````

## Step 8 — Write tests

Create or extend a `test/test-*.jl` file following `.github/instructions/test-writing.instructions.md`:

1. If the algorithm has parameters, test constructor validation.
2. Test the dispatch method with valid inputs.
3. Test that the algorithm integrates correctly when composed inside its target estimator.

## Step 9 — Final checks

Run:

```bash
pre-commit run -a
```

Then in Julia:

```julia-repl
julia> ] activate .
julia> ] test
```

All checks and tests must pass before committing.
