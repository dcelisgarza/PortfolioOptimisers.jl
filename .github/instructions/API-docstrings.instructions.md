---
applyTo: 'src/**/*.jl, docs/**/*.md'
---

# Docstring and Documentation Guidelines for PortfolioOptimisers.jl

## General Guidelines

- Look at how other docstrings are implemented and follow similar patterns.
- Write clear and concise documentation.
- Use consistent terminology and style.
- Include code examples where applicable.
- **All public types, functions, and macros must have docstrings.**

## Grammar

- Use present tense verbs (is, open) instead of past tense (was, opened).
- Write factual statements and direct commands. Avoid hypotheticals like "could" or "would".
- Use active voice where the subject performs the action.
- Write in third person (one, the user) to keep statements consistent.

## Markdown Guidelines

- Use headings to organise content.
- Use bullet points for lists.
- Include links to related resources.
- Use code blocks for code snippets.

---

## DocStringExtensions.jl Macros

Always use `DocStringExtensions.jl` macros where applicable rather than writing boilerplate manually:

| Macro | Use for |
| --- | --- |
| `$(DocStringExtensions.TYPEDEF)` | Docstring header for any `abstract type` or `struct` |
| `$(DocStringExtensions.TYPEDSIGNATURES)` | Docstring header for internal/private functions (auto-generates signature) |
| `$(DocStringExtensions.FIELDS)` | `# Fields` section body — auto-generates field list from inline field docstrings |
| `$(DocStringExtensions.README)` | Module-level docstring (top of main module file) |

Public functions use a **manually written signature** in the docstring header (not `TYPEDSIGNATURES`), because the manual form allows showing default values and grouping overloads clearly.

---

## Documentation Dictionaries

Four dictionaries in `src/01_Base.jl` provide standardised, consistent descriptions. **Always interpolate from them** instead of writing ad-hoc text.

- `arg_dict` — argument descriptions. Use as `$(arg_dict[:key])` in `# Arguments` sections.
- `field_dict` — field descriptions (derived from `arg_dict`). Use as `"$(field_dict[:key])"` in inline field docstrings inside structs.
- `val_dict` — validation rule descriptions. Use as `$(val_dict[:key])` in `## Validation` sections.
- `ret_dict` — return value descriptions. Use as `$(ret_dict[:key])` in `# Returns` sections.
- `math_dict` — LaTeX mathematical notation. Use as `$(math_dict[:key])` in `# Details` sections.

If a needed key is missing, add it to the appropriate dictionary in `01_Base.jl` before writing the docstring.

### Inline field docstrings

Fields in `@concrete` structs are documented inline using `field_dict`:

```julia
@concrete struct Covariance <: AbstractCovarianceEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:malg])"
    alg
    ...
end
```

---

## Section Structure for Types (abstract and concrete)

### Abstract types

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

### Concrete struct types

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this type does.

Optional longer explanation.

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
        # validation
        return new{typeof(field1), typeof(field2)}(field1, field2)
    end
end
function MyType(; field1::Type1 = default1, field2::Type2 = default2)
    return MyType(field1, field2)
end
````

---

## Section Structure for Functions

Public functions use a manually written signature as the docstring header.

````julia
"""
    function_name(
        arg1::Type1,
        arg2::Type2;
        kwarg1::Type3 = default
    ) -> ReturnType

One-sentence description of what this function does.

Longer explanation if needed.

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

Internal/private functions may use `$(DocStringExtensions.TYPEDSIGNATURES)` as the header instead of a manually written signature.

---

## The `# Validation` Section

- Include a `## Validation` sub-section in struct docstrings and a `# Validation` section in function docstrings whenever the function or constructor enforces preconditions.
- Use `val_dict` entries for common validation rules.
- For custom validation, describe the condition clearly: `` `x > 0` ``.

---

## `jldoctest` Examples

- All examples in docstrings must use `jldoctest` blocks so they are testable via `julia --doctest`.
- Output of pretty-printed structs must match exactly what `@define_pretty_show` produces.
- For abstract types with an `# Interfaces` section, the `jldoctest` must demonstrate a complete working implementation of the interface.
- Keep examples minimal but complete enough to be useful.

---

## `docs/src/api/` Markdown Files

Each source file `src/SomeFeature.jl` has a corresponding `docs/src/api/SomeFeature.md`. Every public symbol defined in the source file must be listed under an appropriate heading using the Documenter.jl `@docs` block:

````markdown
## My section heading

```@docs
MyType
my_function
MyAbstractType
```
````

When adding a new symbol, also add it to the corresponding API markdown file.
