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

Optional longer explanation with mathematical notation if needed.

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

---

## Complete Example

The following is a fully worked example covering abstract types, concrete types, and functions. Use it as a reference when writing docstrings.

````julia
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom processes in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a custom process should subtype `MyAbstractCustomProcess`.

# Interfaces

In order to implement a new custom process that can seamlessly work with the library, subtype `MyAbstractCustomProcess`, ensuring that the structure contains all necessary parameters for the custom process, and implement the following methods:

## Custom process interface

### Functions

- `do_process(pr::MyAbstractCustomProcess, b::Real, c::Integer)`: Performs the custom process.

#### Arguments

- `pr`: Custom process.
- `b`: First argument for the custom process.
- `c`: Second argument for the custom process.

#### Returns

- `nothing`.

### Examples

We can create a dummy custom process as follows:

```jldoctest
julia> struct MyNewCustomProcess{T1, T2} <: PortfolioOptimisers.MyAbstractCustomProcess
           alg::T1
           new_param::T2
           function MyNewCustomProcess(alg::MyAbstractCustomProcessAlgorithm, new_param::Symbol)
               return new{typeof(alg), typeof(new_param)}(alg, new_param)
           end
       end

julia> function MyNewCustomProcess(; alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(), new_param::Symbol = :Foo)
           return MyNewCustomProcess(alg, new_param)
        end

julia> function PortfolioOptimisers.do_process(a::MyNewCustomProcess, b::Real, c::Integer)
          println("new custom process: $b $c $(a.sym)")
          do_algorithm(a.alg, c)
          return nothing
       end

julia> do_process(MyNewCustomProcess(), -0.5, 9)
new custom process: -0.5 9 Foo
algorithm 1: 9
```

# Related

- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
abstract type MyAbstractCustomProcess end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom process algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a custom process algorithms should subtype `MyAbstractCustomProcessAlgorithm`.

# Interfaces

In order to implement a new custom process algorithms that can seamlessly work with the library, subtype `MyAbstractCustomProcessAlgorithm`, ensuring that the structure contains all necessary parameters for the custom process algorithm, and implement the following methods:

## Custom process algorithm interface

### Functions

- `do_algorithm(pra::MyAbstractCustomProcessAlgorithm, c::Integer) -> Integer`: Performs the custom process algorithm and returns the result.

#### Arguments

- `pra`: Custom process algorithm.
- `c`: Argument for the custom process algorithm.

#### Returns

- `res::Integer`: The result of the algorithm.

### Examples

We can create a dummy custom process algorithm as follows:

```jldoctest
julia> struct MyNewCustomProcessAlgorithm{T} <: PortfolioOptimisers.MyAbstractCustomProcessAlgorithm
           new_param::T
           function MyNewCustomProcessAlgorithm(new_param::Symbol)
               return new{typeof(new_param)}(new_param)
           end
       end

julia> function MyNewCustomProcessAlgorithm(; new_param::Symbol = :Bar)
           return MyNewCustomProcessAlgorithm(new_param)
        end

julia> function PortfolioOptimisers.do_algorithm(alg::MyNewCustomProcessAlgorithm, c::Integer)
          println("new algorithm: $c $(alg.new_param)")
          return c + 1
       end

julia> do_algorithm(MyNewCustomProcessAlgorithm(), 3)
new algorithm: 3 Bar
4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
abstract type MyAbstractCustomProcessAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Implements my custom process algorithm 1.

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyCustomProcessAlgorithm1 <: MyAbstractCustomProcessAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process algorithm 1.

# Arguments

- `alg::MyCustomProcessAlgorithm1`: The algorithm to perform.
- `c::Integer`: The input integer.

# Returns

- `res::Integer`: The result of the algorithm.

# Details

- Multiplies `c` by 2.
- Prints the result with a custom message.
- Returns the result.

```jldoctest
julia> do_algorithm(MyCustomProcessAlgorithm1(), 3)
algorithm 1: 6
6
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
"""
function do_algorithm(::MyCustomProcessAlgorithm1, c::Integer)
    c = c * 2
    println("algorithm 1: $c")
    return c
end
"""
$(DocStringExtensions.TYPEDEF)

Defines my custom process 1.

# Fields

- `alg::MyAbstractCustomProcessAlgorithm`: The algorithm to use.

# Constructors

    MyConcreteCustomProcess1(;
        alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1()
    ) -> MyConcreteCustomProcess1

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> MyConcreteCustomProcess1()
MyConcreteCustomProcess1
  alg ┴ MyCustomProcessAlgorithm1()
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyConcreteCustomProcess1{T} <: MyAbstractCustomProcess
    alg::T
    function MyConcreteCustomProcess1(alg::MyAbstractCustomProcessAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function MyConcreteCustomProcess1(;
                                  alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1())
    return MyConcreteCustomProcess1(alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process 1.

# Arguments

- `a::MyConcreteCustomProcess1`: The custom process to perform.
- `b::Real`: The first argument.
- `c::Integer`: The second argument.

# Validation

- `b >= 0`: `b` must be non-negative.

# Returns

- `nothing`.

# Details

- Checks `b >= 0` before performing the process.
- Prints a message using the arguments `b` and `c`.
- Calls `do_algorithm` with the algorithm from `a.alg` and `c`.

# Examples

```jldoctest
julia> do_process(MyConcreteCustomProcess1(), 1.0, 2)
Custom process 1: 1.0 + 2
algorithm 1: 4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`MyConcreteCustomProcess1`](@ref)
- [`do_algorithm`](@ref)
"""
function do_process(a::MyConcreteCustomProcess1, b::Real, c::Integer)
    @argcheck(b >= 0, "b must be non-negative")
    println("Custom process 1: $b + $c")
    do_algorithm(a.alg, c)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validates that `val > 2`.

# Arguments

- `val::Real`: The value to validate.

# Returns

- `nothing`.

# Related

- [`MyConcreteCustomProcess2`](@ref)
"""
function assert_val_value(val::Real)
    @argcheck(val >= 2 * one(eltype(val)), "val must be non-negative")
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Defines my custom process 2.

# Fields

- `alg::MyAbstractCustomProcessAlgorithm`: The algorithm to use.
- `val::Real`: The value to use.

# Constructors

    MyConcreteCustomProcess2(;
        alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(),
        val::Real = 2.0
    ) -> MyConcreteCustomProcess2

Keywords correspond to the struct's fields.

## Validation

- `val` is validated via [`assert_val_value`](@ref).

# Examples

```jldoctest
julia> MyConcreteCustomProcess2()
MyConcreteCustomProcess2
  alg ┼ MyCustomProcessAlgorithm1()
  val ┴ 2.0
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`assert_val_value`](@ref)
- [`do_process`](@ref)
- [`do_algorithm`](@ref)
"""
struct MyConcreteCustomProcess2{T1, T2} <: MyAbstractCustomProcess
    alg::T1
    val::T2
    function MyConcreteCustomProcess2(alg::MyAbstractCustomProcessAlgorithm, val::Real)
        return new{typeof(alg), typeof(val)}(alg, val)
    end
end
function MyConcreteCustomProcess2(;
                                  alg::MyAbstractCustomProcessAlgorithm = MyCustomProcessAlgorithm1(),
                                  val::Real = 2.0)
    assert_val_value(val)
    return MyConcreteCustomProcess2(alg, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Performs the custom process 2.

# Arguments

- `a::MyConcreteCustomProcess2`: The custom process to perform.
- `b::Real`: The first argument.
- `c::Integer`: The second argument.

# Returns

- `nothing`.

# Details

- Prints a message using the arguments `a.val`, `b` and `c`.
- Calls `do_algorithm` with the algorithm from `a.alg` and `c`.

# Examples

```jldoctest
julia> do_process(MyConcreteCustomProcess2(), 1.0, 2)
Custom process 2: 2.0 - 1.0 + 2
algorithm 1: 4
```

# Related

- [`MyAbstractCustomProcess`](@ref)
- [`MyAbstractCustomProcessAlgorithm`](@ref)
- [`MyConcreteCustomProcess1`](@ref)
- [`do_algorithm`](@ref)
"""
function do_process(a::MyConcreteCustomProcess2, b::Real, c::Integer)
    println("Custom process 2: $(a.val) - $b + $c")
    do_algorithm(a.alg, c)
    return nothing
end

````
