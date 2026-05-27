---
agent: ask
description: Step-by-step workflow for writing or completing Julia docstrings in PortfolioOptimisers.jl.
---

Follow these steps to write or complete docstrings for symbols in PortfolioOptimisers.jl. Read the referenced files before starting.

## Before you begin

Read the following to understand patterns, conventions, and examples:

- `.github/instructions/julia-docstrings.instructions.md`
- `.github/instructions/julia-source-code.instructions.md`
- `.github/instructions/julia-return-types.instructions.md`
- The dictionaries in `src/01_Base.jl` (`arg_dict`, `field_dict`, `val_dict`, `ret_dict`, `math_dict`).

## Step 1 — Identify what needs documenting

For each symbol (type, function, macro) in scope:

1. Check whether a docstring is present.
2. If present, check it is complete, accurate, and up to date.
3. Note any missing `arg_dict` / `field_dict` / `val_dict` / `ret_dict` / `math_dict` entries needed.

## Step 2 — Add missing dictionary entries

Before writing any docstring, add all required keys to the appropriate dictionary in `src/01_Base.jl`. This ensures interpolation works and descriptions stay consistent across all docstrings.

```julia
arg_dict[:my_key] = "`my_arg::MyType`: Description of the argument."
field_dict[:my_key] = arg_dict[:my_key]
val_dict[:my_key] = "`my_arg > 0`: `my_arg` must be positive."
ret_dict[:my_key] = "`result::MyType`: Description of the return value."
math_dict[:my_key] = "``\\math_expression``: Description of the mathematical concept/variable."
```

## Step 3 — Write abstract type docstrings

Use `$(DocStringExtensions.TYPEDEF)` as the header. Include an `# Interfaces` section that describes the methods a concrete subtype must implement, with argument and return documentation, and a `jldoctest` showing a minimal working implementation.

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this abstract type represents.

All concrete subtypes should subtype `MyAbstractType`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype
`MyAbstractType` and implement the following methods:

## Required method name

  - `method_name(x::MyAbstractType, arg::Type) -> ReturnType`: What the method does.

### Arguments

  - `x`: The concrete subtype instance.
  - `arg`: Description.

### Returns

  - `result::ReturnType`: Description.

### Examples

```jldoctest
julia> struct MyConcreteType <: PortfolioOptimisers.MyAbstractType end
...
```

## Related

  - [`ConcreteSubtype1`](@ref)
  - [`related_function`](@ref)
"""
abstract type MyAbstractType <: AbstractEstimator end
````

## Step 4 — Add mathematical notation (if applicable)

If the type or function has a mathematical formulation, add a `# Mathematical definition` section:

- Place it immediately before `# Fields` for structs, or before `# Arguments` for functions.
- Use `\begin{align}...\end{align}` with `&` alignment markers and `\\` line breaks.
- Put each equation on its own line. Never use `\qquad` to separate distinct equations on one line.
- Use `\boldsymbol` for vectors, `\mathbf` for matrices, `\mathbb` for number domains, `\mathcal` for sets, `\intercal` for transpose.
- Use `\underset{\boldsymbol{x}}{\min}` for optimisation objectives.
- Immediately after the last math block, add a `Where:` bullet list defining every symbol.
- Interpolate `$(math_dict[:key])` for standardised variables. Add missing keys to `src/01_Base.jl` first.

See the `## Mathematical Notation` section in `.github/instructions/julia-docstrings.instructions.md` for the full worked example.

## Step 5 — Write concrete struct docstrings

Use `$(DocStringExtensions.TYPEDEF)` as the header. Document fields inline using `field_dict`. Include a `# Constructors` section with the keyword-arg signature, a `## Validation` subsection listing all preconditions, and a `jldoctest` showing default construction.

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this type does.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyType(;
        field1::Type1 = default1,
        field2::Type2 = default2
    ) -> MyType

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:key1])
  - $(val_dict[:key2])

# Examples

```jldoctest
julia> MyType()
MyType
  field1 ┴ default1
```

## Related

  - [`AbstractMyType`](@ref)
  - [`related_function`](@ref)
"""
@concrete struct MyType <: AbstractMyType
    "$(field_dict[:key1])"
    field1
    "$(field_dict[:key2])"
    field2
    function MyType(field1::Type1, field2::Type2)
        @argcheck condition DomainError(field1, "message")
        return new{typeof(field1), typeof(field2)}(field1, field2)
    end
end
function MyType(; field1::Type1 = default1, field2::Type2 = default2)
    return MyType(field1, field2)
end
````

## Step 6 — Write public function docstrings

Use a **manually written** signature as the header (not `TYPEDSIGNATURES`) so that default values and overloads are shown clearly. Include `# Arguments`, `# Validation` (if applicable), `# Returns`, `# Details` (if applicable), and `# Examples`.

````julia
"""
    function_name(
        arg1::Type1,
        arg2::Type2;
        kwarg1::Type3 = default
    ) -> ReturnType

One-sentence description of what this function does.

# Arguments

  - $(arg_dict[:key1])
  - $(arg_dict[:key2])
  - `kwarg1::Type3 = default`: Description.

# Validation

  - $(val_dict[:key])

# Returns

  - $(ret_dict[:key])

# Details

  - Additional implementation notes.

# Examples

```jldoctest
julia> function_name(...)
...
```

## Related

  - [`RelatedType`](@ref)
  - [`related_function`](@ref)
"""
function function_name(arg1::Type1, arg2::Type2; kwarg1::Type3 = default)
    ...
end
````

## Step 7 — Write internal/private function docstrings

Use `$(DocStringExtensions.TYPEDSIGNATURES)` as the header. Include `# Arguments`, `# Returns`, and `# Examples` sections. The `# Validation` section is only needed if the function enforces preconditions.

````julia
"""
$(DocStringExtensions.TYPEDSIGNATURES)

One-sentence description.

# Arguments

  - `arg1::Type1`: Description.

# Returns

  - `result::ReturnType`: Description.

# Examples

```jldoctest
julia> _internal_fn(...)
...
```

## Related

- [`PublicType`](@ref)
"""
function _internal_fn(arg1::Type1)::ReturnType
    ...
end

````

## Step 8 — Add to API docs

For every new or updated public symbol, ensure it is listed in the corresponding `docs/src/api/*.md` file under an appropriate heading:

````markdown
```@docs
MyType
my_function
MyAbstractType
```
````

The correspondence is: `src/SomeFeature.jl` → `docs/src/api/SomeFeature.md`.

## Step 9 — Final checks

Run the full pre-commit, test, and doctest suite following `.github/prompts/pre-commit-and-test.prompt.md`.

All three steps must pass before committing.
